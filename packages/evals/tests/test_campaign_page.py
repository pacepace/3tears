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
from datetime import datetime
from collections.abc import Callable
from html.parser import HTMLParser
from typing import Any, Literal

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
from threetears.evals.analysis.bundle.schema import QuestionAfterRuns
from threetears.evals.analysis.lenses.frontier import FrontierCostDecision, FrontierCostTie, FrontierVerdict
from threetears.evals.analysis.report.campaign_page import CHOICE_NOTE, NO_CALL, NO_RECOMMENDATION
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
from threetears.evals.kernel.declaration import BarOverride, CampaignCall, Floor, OverriddenBar, RateDrop
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

#: What a flag says was set after the runs.
QuestionAfterRunsLate = Literal["question", "control"]

#: When the late aspect's question was asked: long after its runs.
_ASKED_LATE = "2099-01-01T00:00:00+00:00"


def _paged(stamp: str) -> str:
    """An ISO-8601 instant as the page dates it: ``1 Jan 2099``."""
    moment = datetime.fromisoformat(stamp)
    return f"{moment.day} {moment:%b %Y}"


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


async def test_an_aspect_asked_after_its_runs_reads_as_asked_and_the_page_says_when_under_the_question() -> None:
    """Owner ruling (2026-10-21): disclosed, not enforced. Its row reads as tone's, asked in time, does: no lead."""
    page = await _page("billing", "late")
    first = _first_screen_markdown(page)
    late_bundle = page.aspects[1].report.bundle
    assert late_bundle.design_frozen_at is not None
    began = _paged(late_bundle.design_frozen_at)
    said = f"The question was set on 1 Jan 2099, after the runs on late began on {began}."
    assert said in first, "billing was asked in time, so the line names the aspect it holds on"
    assert first.index(QUESTION) < first.index(said)
    assert "a lead" not in first
    (late,) = page.readings.on_aspect("late")
    assert late.confirmatory and late.reading == "better"
    (tone,) = (await _page("billing", "tone")).readings.on_aspect("tone")
    assert (late.reading, late.change, late.gain_interval) == (tone.reading, tone.change, tone.gain_interval)
    assert "was set on" not in _first_screen_markdown(await _page("billing", "tone")), "asked in time: nothing said"


async def test_a_question_with_no_recorded_date_is_said_to_be_possibly_after_the_runs() -> None:
    def undated(bundle: AnalysisContextBundle) -> AnalysisContextBundle:
        late = bundle.reading_scope.question_after_runs
        if late is None:
            return bundle
        scope = bundle.reading_scope.model_copy(update={"question_after_runs": late.model_copy(update={"at": None})})
        return bundle.model_copy(update={"reading_scope": scope})

    page = await _page("billing", "late", bundle=undated)
    first = _first_screen_markdown(page)
    assert "The question carries no date, so it may have been set after the runs on late began on " in first


#: When billing's question is flagged as set below: after its runs began, which these tests take as given.
_SET = "2099-01-01T00:00:00+00:00"

#: Later than :data:`_SET`: something declared after the late question.
_LATER = "2099-06-01T00:00:00+00:00"


def _flagged(
    late: QuestionAfterRunsLate = "question", at: str | None = _SET, *, margin_at: str | None = _SET
) -> Callable[[AnalysisContextBundle], AnalysisContextBundle]:
    """Billing's bundle with what was set after its runs flagged at ``at``, its question worded then, and its margin
    declared at ``margin_at``."""

    def change(bundle: AnalysisContextBundle) -> AnalysisContextBundle:
        if bundle.aspect != "billing":
            return bundle
        assert bundle.declared is not None and bundle.design_frozen_at is not None
        flag = QuestionAfterRuns(late=late, at=at, first_run_at=bundle.design_frozen_at)
        scope = bundle.reading_scope.model_copy(update={"question_after_runs": flag})
        declared = bundle.declared.model_copy(update={"question_declared_at": _SET, "margin_declared_at": margin_at})
        return bundle.model_copy(update={"reading_scope": scope, "declared": declared})

    return change


async def _billing_floor_at(stamp: str | None) -> list[Floor]:
    (floor,) = (await _aspect("billing")).campaign.floors
    return [floor.model_copy(update={"declared_at": stamp})]


async def test_a_margin_and_a_floor_added_after_a_question_set_late_are_still_said() -> None:
    """The line under the question says what was declared with it; what came later is said as any late stamp is."""
    later = await _page("billing", "tone", bundle=_flagged(margin_at=_LATER), floors=await _billing_floor_at(_LATER))
    first = _first_screen_markdown(later)
    assert "The question was set on 1 Jan 2099, after the runs on billing began on " in first
    assert "The margin on billing was declared after its first result." in first
    assert (
        "The floor that no pass rate drops by more than 10 points on billing was declared after its first result."
        in first
    )
    assert "worded after" not in first, "worded when it was set: the line under the question says it"

    with_it = await _page("billing", "tone", bundle=_flagged(), floors=await _billing_floor_at(_SET))
    assert "declared after its first result" not in _first_screen_markdown(with_it), "declared with the question"


async def test_an_undated_question_or_a_late_control_holds_no_late_stamp_back() -> None:
    """Neither orders the margin against the question: a question with no date orders nothing, and a control named
    late dates only itself, so a margin later than the first result is said either way."""
    for flag in (_flagged("question", None), _flagged("control", _SET)):
        first = _first_screen_markdown(await _page("billing", "tone", bundle=flag))
        assert "The margin on billing was declared after its first result." in first


async def test_a_control_named_after_the_first_result_is_said_as_the_control_not_the_question() -> None:
    first = _first_screen_markdown(await _page("billing", "tone", bundle=_flagged("control", _SET)))
    assert "The control was named on 1 Jan 2099, after the runs on billing began on " in first
    assert "The question was set" not in first
    undated = _first_screen_markdown(await _page("billing", "tone", bundle=_flagged("control", None)))
    assert (
        "The control was named at a time that cannot be ordered, so it may have been named after the runs on" in undated
    )


def test_two_times_on_one_day_carry_their_times_of_day() -> None:
    same = QuestionAfterRuns(late="question", at="2026-10-14T15:00:00+00:00", first_run_at="2026-10-14T09:30:00+00:00")
    assert same.page_dates() == ("14 Oct 2026 at 15:00 UTC", "14 Oct 2026 at 09:30 UTC")
    apart = QuestionAfterRuns(late="question", at="2026-10-20T15:00:00+00:00", first_run_at="2026-10-14T09:30:00+00:00")
    assert apart.page_dates() == ("20 Oct 2026", "14 Oct 2026")
    undated = QuestionAfterRuns(late="question", at=None, first_run_at="2026-10-14T09:30:00+00:00")
    assert undated.page_dates() == (None, "14 Oct 2026")


def test_each_time_is_dated_in_its_own_offset_as_the_rest_of_the_page_dates_it() -> None:
    """The Methods row dates the design's freeze in the stamp's own offset; the line under the question dates the same
    instant the same way, so one page never prints one instant as two days. Where the later time falls on no later day
    than the earlier in their own offsets, both carry their times of day and offsets, so it never reads as earlier."""
    began = "2026-10-15T01:00:00+00:00"
    apart = QuestionAfterRuns(late="question", at="2026-10-16T23:30:00-05:00", first_run_at=began)
    assert apart.page_dates() == ("16 Oct 2026", "15 Oct 2026")
    # 04:30 UTC on 15 Oct, after the runs; in its own offset, still 14 Oct.
    crossed = QuestionAfterRuns(late="question", at="2026-10-14T23:30:00-05:00", first_run_at=began)
    assert crossed.page_dates() == ("14 Oct 2026 at 23:30 UTC-05:00", "15 Oct 2026 at 01:00 UTC")


@pytest.mark.parametrize("field", ["at", "first_run_at"])
@pytest.mark.parametrize("stamp", ["sometime in October", "2026-10-14T09:30:00"])
def test_a_flag_whose_time_cannot_be_ordered_is_refused_not_printed_raw(field: str, stamp: str) -> None:
    times = {"at": "2026-10-20T15:00:00+00:00", "first_run_at": "2026-10-14T09:30:00+00:00", field: stamp}
    with pytest.raises(ValidationError, match=rf"the late-question flag's {field} is stamped {re.escape(repr(stamp))}"):
        QuestionAfterRuns(late="question", **times)
    flag = QuestionAfterRuns(late="question", at="2026-10-20T15:00:00+00:00", first_run_at="2026-10-14T09:30:00+00:00")
    assert flag.page_dates() == ("20 Oct 2026", "14 Oct 2026"), "the same flag with ordered times is read"


@pytest.mark.parametrize(
    ("flag", "said"),
    [
        (None, "question declared before the runs"),
        (("question", _SET), "question declared after the runs began"),
        (("question", None), "question undated, so possibly declared after the runs began"),
        (("control", _SET), "question declared before the runs, control named after they began"),
        (("control", None), "question declared before the runs, control named at a time that cannot be ordered"),
    ],
)
async def test_the_methods_name_the_design_as_the_line_under_the_question_does(
    flag: tuple[QuestionAfterRunsLate, str | None] | None, said: str
) -> None:
    bundle = (await _aspect("billing")).bundle
    if flag is not None:
        bundle = _flagged(*flag)(bundle)
    report = build_code_only_report(bundle, assembled_at=ASSEMBLED_AT)
    (methods,) = [block for block in report.blocks if isinstance(block, TableBlock) and block.name == "methods"]
    (design,) = [str(row["value"].value) for row in methods.rows if row["item"].value == "Design"]
    assert design.startswith(said + ", frozen ")


async def test_the_page_json_carries_each_aspect_s_flag_beside_its_readings() -> None:
    page = await _page("billing", "tone", bundle=_flagged())
    aspects = page.json_document()["aspects"]
    assert isinstance(aspects, list)
    billing = (await _aspect("billing")).bundle
    assert [aspect["question_after_runs"] for aspect in aspects] == [
        {"late": "question", "at": _SET, "first_run_at": billing.design_frozen_at},
        None,
    ]


async def test_the_line_saying_when_the_question_was_set_shows_no_field_name() -> None:
    page = await _page("billing", "late")
    (said,) = [
        block.text
        for block in page.blocks
        if isinstance(block, DisclosureBlock) and block.section == "header" and block.source == "timing"
    ]
    assert "_" not in said and "after_runs" not in campaign_report_html(page)


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


#: A summary sentence the style pass keeps.
_KEPT = "Production stays, since billing is where refunds are lost."


async def test_the_summary_is_held_to_the_style_pass_like_any_of_the_memo_s_words() -> None:
    """A sentence restating a measured number is dropped from the summary as from a finding's body."""
    page = await _page(
        "billing",
        "tone",
        memo=_memo().model_copy(update={"summary": f"Production passed 54 of 60 billing trials. {_KEPT}"}),
    )
    (summary,) = [block for block in page.blocks if isinstance(block, TextBlock) and block.role == "summary"]
    assert summary.body == _KEPT
    methods = [block.text for block in page.blocks if block.section == "methods"]
    assert any("restated a measured number" in text for text in methods)


async def test_a_summary_the_style_pass_would_drop_whole_is_kept_as_written() -> None:
    """The summary is one sentence, so a pass that would leave none of it keeps it as written, as a finding's title is:
    the page still opens on the writer's read across the aspects."""
    restated = "Production passed 54 of 60 billing trials, so it stays."
    page = await _page("billing", "tone", memo=_memo().model_copy(update={"summary": restated}))
    (summary,) = [block for block in page.blocks if isinstance(block, TextBlock) and block.role == "summary"]
    assert summary.body == restated
    assert _opening(page)[0] == ("text", "summary")
    # The page prints it, so the methods never say it was left to the tables.
    methods = [block.text for block in page.blocks if block.section == "methods"]
    assert not any("restated a measured number" in text for text in methods)


async def test_a_title_the_style_pass_would_drop_whole_is_printed_listed_as_breaking_a_rule_and_not_counted_dropped() -> (
    None
):
    """A finding keeps its headline as written where the pass would leave none of it, so the methods list the rule it
    breaks and never say the tables said it instead."""
    memo = _memo()
    banned = "The verdict rests on billing."
    finding = memo.findings[0].model_copy(update={"title": banned})
    page = await _page("billing", "tone", memo=memo.model_copy(update={"findings": [finding]}))
    (title,) = [block for block in page.blocks if isinstance(block, TextBlock) and block.role == "finding_title"]
    assert title.body == banned
    methods = [block.text for block in page.blocks if block.section == "methods"]
    assert not any("restated a measured number" in text for text in methods)
    assert any("banned" in text and banned in text for text in methods)


async def test_a_summary_proposing_a_sample_no_code_computed_is_dropped_and_listed() -> None:
    """Kept as written never reaches a sample size or cost no projection states: that summary is dropped, as any such
    sentence is, and the methods list it."""
    proposing = "Run 200 more billing cases for about $40 to settle it."
    page = await _page("billing", "tone", memo=_memo().model_copy(update={"summary": proposing}))
    assert not [block for block in page.blocks if isinstance(block, TextBlock) and block.role == "summary"]
    methods = [block.text for block in page.blocks if block.section == "methods"]
    assert any("a sample size or cost no code computed" in text and "200 more" in text for text in methods)


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
