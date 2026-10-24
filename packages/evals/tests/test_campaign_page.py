"""A campaign's page: code's readings across its aspects on the first screen, then one fold per aspect.

The fixture is two real runs read as one campaign of two aspects: on billing the candidate regresses (it passes six of
twenty cases to production's eighteen) and breaks the 10-point floor declared there; on tone it improves (nineteen to
six). Both readings are on the first screen, the broken floor first of all; the writer's summary across the aspects
and its recommendation, when there is a memo, come next, then code's table, then the person's call. Each aspect's own
page sits in its fold.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from html.parser import HTMLParser
from typing import Any

import pytest
from pydantic import ValidationError

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
from threetears.evals.analysis.bundle.schema import CriterionKind, LateCriterion
from threetears.evals.analysis.lenses.frontier import FrontierCostDecision, FrontierCostTie, FrontierVerdict
from threetears.evals.analysis.report.campaign_page import CHOICE_NOTE, NO_CALL, NO_RECOMMENDATION, campaign_verdict
from threetears.evals.analysis.report.model import (
    CAMPAIGN_JSON_SHAPE,
    CampaignAspectFold,
    DisclosureBlock,
    TableBlock,
    TextBlock,
)
from threetears.evals.analysis.report.words import FIRST_SCREEN_BANNED, NO_RECOMMENDATION_WORDS
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

#: When the late aspect was asked, its margin set and its control named: long after its readings were visible.
_ASKED_LATE = "2099-01-01T00:00:00+00:00"


async def _aspect(label: str) -> MeasuredAspect:
    """Each fixture aspect, measured once for the module: billing regresses past its floor, tone improves, steady
    holds within a 20-point floor and margin (both arms pass every case, which on twenty cases bounds the drop at 19
    points), and late improves on a question asked only after its runs."""
    if label not in _MEASURED:
        if label == "billing":
            _MEASURED[label] = await measured_aspect(
                "billing", production=18, candidate=6, floors=(RateDrop(max_points=10),)
            )
        elif label == "tone":
            _MEASURED[label] = await measured_aspect("tone", production=6, candidate=19)
        elif label == "steady":
            _MEASURED[label] = await measured_aspect(
                "steady", production=20, candidate=20, margin=20, floors=(RateDrop(max_points=20),)
            )
        else:
            _MEASURED[label] = await measured_aspect(label, production=6, candidate=19, asked_at=_ASKED_LATE)
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
        summary=_SUMMARY,
        next=[
            NextStep(
                title="Fix the billing prompt and run again.", why="It is the only loss.", leverage="high", lever=""
            )
        ],
    )


#: The test memo's read across its aspects.
_SUMMARY = "v2 wins on tone and loses billing, where refunds are lost, so production stays."


def _opening(page: CampaignReport) -> list[tuple[str, str]]:
    """What the page says after its warnings and its question, up to the first table: each text block by its role,
    each line by its source and text."""
    said: list[tuple[str, str]] = []
    for block in page.blocks:
        if isinstance(block, TableBlock):
            break
        if isinstance(block, TextBlock) and block.role != "override":
            said.append(("text", block.role))
        elif isinstance(block, DisclosureBlock) and block.source not in {"bar", "floor", "guardrails", "basis"}:
            said.append((block.source, block.text))
    return said


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


async def test_the_first_screen_runs_in_order_floor_question_summary_recommendation_table_call() -> None:
    """A change with a recommendation reads its summary, then the recommendation, then the aspect table."""
    page = await _page("billing", "tone", memo=_memo())
    first = _first_screen_markdown(page)
    order = [
        first.index("**Floor broken on billing"),
        first.index(f'"{QUESTION}"'),
        first.index(_SUMMARY),
        first.index(f"**Recommended:** {CONTROL_LABEL} (high confidence)"),
        first.index("- The billing floor is broken"),
        first.index("**By aspect**"),
        first.index(NO_CALL),
    ]
    assert order == sorted(order)
    assert page.basis == "analysis" and page.finding_count == 1
    header = _header_html(page)
    order = [
        header.index('role="alert"'),
        header.index("<h1>"),
        header.index('data-role="summary"'),
        header.index('data-role="recommendation"'),
        header.index('data-table="aspects"'),
    ]
    assert order == sorted(order)
    assert _opening(page) == [("text", "summary"), ("text", "recommendation"), ("text", "reason")]


async def test_a_code_only_page_has_no_summary_and_says_no_recommendation_was_written_before_the_table() -> None:
    page = await _page("billing", "tone")
    first = _first_screen_markdown(page)
    assert first.index(f'"{QUESTION}"') < first.index(NO_RECOMMENDATION) < first.index("**By aspect**")
    assert _opening(page)[0] == ("memo", NO_RECOMMENDATION)


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


#: When a criterion is flagged as set below: after its aspect's readings were visible, which these tests take as given.
_SET = "2099-01-01T00:00:00+00:00"


def _set_late(
    *late: tuple[CriterionKind, str | None, str | None], on: Sequence[str] = ("billing",)
) -> Callable[[AnalysisContextBundle], AnalysisContextBundle]:
    """The bundles of the aspects ``on``, each with ``late`` (kind, name, time) as its criteria set after its
    readings were visible."""

    def change(bundle: AnalysisContextBundle) -> AnalysisContextBundle:
        if bundle.aspect not in on:
            return bundle
        criteria = [LateCriterion(criterion=kind, name=name, set_at=at) for kind, name, at in late]
        scope = bundle.reading_scope.model_copy(update={"late_criteria": criteria})
        return bundle.model_copy(update={"reading_scope": scope})

    return change


def _timing(page: CampaignReport) -> list[str]:
    """The first screen's lines naming criteria set after their readings, in order."""
    return [
        block.text
        for block in page.blocks
        if isinstance(block, DisclosureBlock) and block.section == "header" and block.source == "timing"
    ]


async def test_a_margin_late_on_some_aspects_is_one_line_naming_them() -> None:
    page = await _page("billing", "tone", bundle=_set_late(("margin", None, _SET), on=("tone",)))
    assert _timing(page) == ["The margin on tone was set on 1 Jan 2099, after its readings were visible."]
    # Billing's margin came before its first result: nothing is said of it.
    assert "billing was set" not in _first_screen_markdown(page)


async def test_an_aspect_set_after_its_readings_reads_as_asked_and_each_criterion_is_said_once() -> None:
    """Owner rulings (2026-10-21, 2026-10-23): disclosed, not enforced. Its row reads as tone's, set in time, does: no
    lead. Each criterion set late is named once, with its time, under the question, and no line names the question or
    when the runs began."""
    page = await _page("billing", "late")
    first = _first_screen_markdown(page)
    said = _timing(page)
    assert said == [
        "The aspect late was declared on 1 Jan 2099, after its readings were visible.",
        "The control was named on 1 Jan 2099, after the readings on late were visible.",
        "The margin on late was set on 1 Jan 2099, after its readings were visible.",
    ], "billing was set in time, so each line names the aspect it holds on"
    assert all(first.index(QUESTION) < first.index(line) for line in said)
    assert "a lead" not in first
    assert "The question" not in " ".join(said) and "began" not in first and "after the runs" not in first
    (late,) = page.readings.on_aspect("late")
    assert late.confirmatory and late.reading == "better"
    (tone,) = (await _page("billing", "tone")).readings.on_aspect("tone")
    assert (late.reading, late.change, late.gain_interval) == (tone.reading, tone.change, tone.gain_interval)
    assert _timing(await _page("billing", "tone")) == [], "set in time: nothing said"


async def test_a_criterion_with_no_recorded_time_says_so_and_never_guesses() -> None:
    page = await _page(
        "billing", "tone", bundle=_set_late(("aspect", None, None), ("control", None, None), ("margin", None, None))
    )
    assert _timing(page) == [
        "When the aspect billing was declared is not recorded.",
        "When the control was named is not recorded.",
        "When the margin on billing was set is not recorded.",
    ]


async def test_a_bar_and_a_floor_set_after_their_readings_are_each_named_by_what_they_hold() -> None:
    (floor,) = (await _aspect("billing")).campaign.floors
    page = await _page("billing", "tone", bundle=_set_late(("bar", "cost_usd", _SET), ("floor", floor.key, _SET)))
    bar, floored = _timing(page)
    assert bar.startswith("The ") and bar.endswith(
        " bar on billing was set on 1 Jan 2099, after its readings were visible."
    )
    assert floored == (
        "The floor that no pass rate drops by more than 10 points on billing was set on 1 Jan 2099, after its "
        "readings were visible."
    )


async def test_a_criterion_late_on_every_aspect_is_one_line_never_one_per_aspect() -> None:
    page = await _page(
        "billing", "tone", bundle=_set_late(("aspect", None, _SET), ("margin", None, _SET), on=("billing", "tone"))
    )
    assert _timing(page) == [
        "Every aspect was declared on 1 Jan 2099, after its readings were visible.",
        "The margins on every aspect were set on 1 Jan 2099, after their readings were visible.",
    ]
    control = await _page("billing", "tone", bundle=_set_late(("control", None, _SET), on=("billing", "tone")))
    assert _timing(control) == ["The control was named on 1 Jan 2099, after every aspect's readings were visible."]


async def test_the_one_aspect_page_says_its_late_criteria_and_the_campaign_s_fold_does_not_repeat_them() -> None:
    page = await _page("billing", "late")
    own = build_code_only_report((await _aspect("late")).bundle, assembled_at=ASSEMBLED_AT)
    own_timing = [
        block.text
        for block in own.blocks
        if isinstance(block, DisclosureBlock) and block.section == "header" and block.source == "timing"
    ]
    assert own_timing == _timing(page)
    fold = next(fold for fold in page.aspects if fold.aspect == "late")
    assert not [block for block in fold.head if isinstance(block, DisclosureBlock) and block.source == "timing"]


def test_a_late_criterion_is_dated_in_its_own_offset_as_the_rest_of_the_page_dates_it() -> None:
    """The Methods row dates the design's freeze in the stamp's own offset; a late criterion's line dates its instant
    the same way, so one page never prints one instant as two days."""
    assert LateCriterion(criterion="margin", set_at="2026-10-16T23:30:00-05:00").page_date() == "16 Oct 2026"
    assert LateCriterion(criterion="margin", set_at="2026-10-14T23:30:00-05:00").page_date() == "14 Oct 2026"
    assert LateCriterion(criterion="margin", set_at=None).page_date() is None


@pytest.mark.parametrize("stamp", ["sometime in October", "2026-10-14T09:30:00"])
def test_a_late_criterion_whose_time_cannot_be_ordered_is_refused_not_printed_raw(stamp: str) -> None:
    with pytest.raises(ValidationError, match=rf"a late criterion's set_at is stamped {re.escape(repr(stamp))}"):
        LateCriterion(criterion="margin", set_at=stamp)
    assert LateCriterion(criterion="margin", set_at="2026-10-20T15:00:00+00:00").page_date() == "20 Oct 2026"


@pytest.mark.parametrize("late", [False, True])
async def test_the_methods_name_the_design_without_its_timing(late: bool) -> None:
    bundle = (await _aspect("billing")).bundle
    if late:
        bundle = _set_late(("aspect", None, _SET))(bundle)
    report = build_code_only_report(bundle, assembled_at=ASSEMBLED_AT)
    (methods,) = [block for block in report.blocks if isinstance(block, TableBlock) and block.name == "methods"]
    (design,) = [str(row["value"].value) for row in methods.rows if row["item"].value == "Design"]
    assert design.startswith("question declared, frozen ")


async def test_the_page_json_carries_each_aspect_s_late_criteria_beside_its_readings() -> None:
    page = await _page("billing", "tone", bundle=_set_late(("margin", None, _SET)))
    aspects = page.json_document()["aspects"]
    assert isinstance(aspects, list)
    assert CAMPAIGN_JSON_SHAPE == 4
    assert [aspect["late_criteria"] for aspect in aspects] == [
        [{"criterion": "margin", "name": None, "set_at": _SET}],
        [],
    ]


async def test_the_lines_naming_late_criteria_show_no_field_name() -> None:
    page = await _page("billing", "late")
    said = _timing(page)
    assert said and not any("_" in line for line in said)
    assert "late_criteria" not in campaign_report_html(page)


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


async def test_an_unclear_row_s_cases_show_it_no_worse_than_the_margin_and_not_better() -> None:
    page = await _page("billing", "tone", margins={"billing": 70.0})
    settle = next(block for block in page.blocks if isinstance(block, TableBlock) and block.name == "to_settle")
    assert settle.title == "To show no worse than the margin"
    note = page.blocks[page.blocks.index(settle) + 1]
    assert isinstance(note, DisclosureBlock)
    assert "would not show the arm better, so its row stays unclear unless its gain clears too" in note.text


async def test_a_row_no_worse_than_the_margin_reads_so_beside_its_held_floor_and_has_nothing_to_settle() -> None:
    page = await _page("steady", "tone")
    (row,) = page.readings.on_aspect("steady")
    (status,) = page.readings.floors
    assert row.reading == "within_margin" and status.status == "held"
    table = next(block for block in page.blocks if isinstance(block, TableBlock) and block.name == "aspects")
    readings = {cells["aspect"].text: cells["reading"].text for cells in table.rows}
    assert readings == {"steady": "no worse than the margin", "tone": "shown better"}
    # The headline row and the floor read one interval, so neither contradicts the other, and nothing is to settle.
    assert not any(isinstance(block, TableBlock) and block.name == "to_settle" for block in page.blocks)
    first = _first_screen_markdown(page)
    assert (
        "A row no worse than the margin reads its interval at 95% on its own, as a floor does: a candidate is no "
        "worse only where it holds on every asked aspect, which needs no correction." in first
    )
    # The one corrected row is tone's, read in v2's family of two at 97.5%, so it is not called a floor's level; the
    # error-rate line is said whatever the rows read.
    assert "The interval is read at 97.5%, corrected across each arm's aspects." in first
    assert "Each arm's gains and its drops are each shown wrongly at most 2.5% of the time." in first
    assert "such an interval clearing zero is not a gain shown" not in first
    markdown = campaign_report_markdown(page)
    steady = markdown[markdown.index("## steady") : markdown.index("## tone")]
    assert "v2 no worse than the margin." in steady
    assert (
        campaign_verdict(page.readings, ["steady", "tone"])
        == "steady: v2 no worse than the margin; tone: v2 shown better"
    )


async def test_a_page_whose_every_row_is_no_worse_still_states_the_error_rate_and_that_nothing_corrects_across_arms() -> (
    None
):
    def all_within(readings: CampaignReadings) -> CampaignReadings:
        rows = [
            row.model_copy(update={"reading": "within_margin", "within_interval": (1.3, 62.7), "projection": None})
            for row in readings.readings
        ]
        return readings.model_copy(update={"readings": rows + [rows[0].model_copy(update={"arm": "v3"})]})

    page = await _page("steady", "tone", readings=all_within)
    first = _first_screen_markdown(page)
    assert "The interval is read at" not in first and "Each interval is read at" not in first
    assert "Each arm's gains and its drops are each shown wrongly at most 2.5% of the time." in first
    assert (
        "Nothing corrects across arms, so with several arms one is shown better or worse by chance more often." in first
    )
    assert (
        "Its gain is still read corrected across the arm's aspects, so such an interval clearing zero is not a gain "
        "shown." in first
    )


async def test_a_floor_held_beside_a_row_left_unclear_says_the_two_read_different_points() -> None:
    # The same interval against a 10-point margin is unclear while the 20-point floor holds.
    page = await _page("steady", margins={"steady": 10.0})
    (row,) = page.readings.on_aspect("steady")
    (status,) = page.readings.floors
    assert row.reading == "unclear" and status.status == "held"
    assert (
        "The floor held on steady for v2 reads its pass rate against 20 points, while its row reads the pass rate "
        "against the 10-point margin." in _first_screen_markdown(page)
    )
    settle = next(block for block in page.blocks if isinstance(block, TableBlock) and block.name == "to_settle")
    assert [cells["aspect"].text for cells in settle.rows] == ["steady"]
    note = page.blocks[page.blocks.index(settle) + 1]
    assert isinstance(note, DisclosureBlock)
    assert "margin, read at 95%. A floor on the same rate and points gives the same count." in note.text


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
    # The writer's summary and recommendation, then the pairs; no control-based table.
    assert first.index(_SUMMARY) < first.index("**Recommended:** beta (high confidence)")
    assert first.index("**Recommended:** beta (high confidence)") < first.index("**By aspect, pair by pair**")
    assert "**By aspect**\n" not in first
    markdown = campaign_report_markdown(page)
    assert "alpha ahead of beta." in markdown and "alpha and beta not separated." in markdown


def _recommending_nothing(summary: str) -> AuthoredCampaignAnalysis:
    """The test memo recommending no arm, with ``summary`` read across its aspects."""
    return _memo().model_copy(update={"recommendation": [], "aspect_summaries": [], "summary": summary})


async def test_a_choice_with_no_clear_winner_opens_on_its_summary_then_says_so_then_the_pairs() -> None:
    summary = "Neither arm pulls ahead on billing or refunds, so cost is the only thing left to choose on."
    page = await _kind_page("billing", "refunds", memo=_recommending_nothing(summary))
    assert page.readings.kind == "choice" and page.recommendation is None
    assert _opening(page) == [("text", "summary"), ("memo", NO_RECOMMENDATION_WORDS["choice"])]
    first = _first_screen_markdown(page)
    assert first.index(summary) < first.index("No clear winner") < first.index("**By aspect, pair by pair**")
    header = _header_html(page)
    assert header.index("<h1>") < header.index('data-role="summary"') < header.index('data-table="pairs"')


async def test_a_measurement_opens_after_its_warning_on_its_summary_then_its_measures() -> None:
    summary = "Alpha passes most cases on solo and costs too much a reply to meet its bar."
    page = await _kind_page("solo", memo=_recommending_nothing(summary))
    assert page.readings.kind == "measurement"
    assert _opening(page) == [("text", "summary")]
    first = _first_screen_markdown(page)
    assert first.startswith("**Bar missed on solo:")
    assert first.index("# ") < first.index(summary) < first.index("**By aspect, measure by measure**")
    header = _header_html(page)
    assert header.index('role="alert"') < header.index("<h1>") < header.index('data-role="summary"')
    assert header.index('data-role="summary"') < header.index('data-table="measures"')


async def test_the_memo_s_words_are_printed_as_written_in_any_language() -> None:
    """Code checks code, and a model's prose is a model's to check (owner, 2026-10-24): the page reads none of the memo's
    words. A sentence an English word list once dropped (a restated count, a discount by data age, a proposed sample)
    and a sentence in French reach the page as written, and the methods list nothing the memo said."""
    memo = _memo()
    summary = "La production reste : la facturation perd des remboursements. Run 200 more billing cases for $40."
    body = "Production passed 54 of 60 billing trials. The runs predate the campaign, so read the tone gain as provisional."
    finding = memo.findings[0].model_copy(update={"title": "Le ton progresse sur v2", "body": body})
    page = await _page("billing", "tone", memo=memo.model_copy(update={"summary": summary, "findings": [finding]}))
    said = {block.role: block.body for block in page.blocks if isinstance(block, TextBlock)}
    assert (said["summary"], said["finding_title"], said["finding_body"]) == (summary, "Le ton progresse sur v2", body)
    assert not [
        block
        for block in page.blocks
        if isinstance(block, DisclosureBlock) and block.section == "methods" and block.source == "memo"
    ]


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
    assert first.index("**Recommended:** beta") < pairs < why


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
